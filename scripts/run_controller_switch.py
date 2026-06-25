#!/usr/bin/env python3
"""Does the navigation controller switch in the dark? (behaviour only)

Headline test: on conflict junctions — where the egocentric (alternation/forward)
rule and the allocentric (least-recently-visited) rule predict different arms —
what fraction does the animal follow the allocentric rule, light vs dark? If the
world-based rule is vision-gated, that fraction should drop from light to dark.

Also reports each rule's standalone choice-prediction accuracy per condition.

No assumption about RSP function; no neural data. See
docs/plan-controller-switch-behaviour.md.

Usage
-----
    python scripts/run_controller_switch.py            # full run
    python scripts/run_controller_switch.py --limit 3  # quick
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.maze.choice_models import (  # noqa: E402
    conflict_follow_rate,
    extract_choice_events,
    rule_accuracies,
)
from hm2p.maze.discretize import cell_sequence, discretize_position_fast  # noqa: E402
from hm2p.maze.topology import build_rose_maze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("controller_switch")

BUCKET = "hm2p-derivatives"
SPEED_THRESHOLD = 2.5
MAZE = build_rose_maze()
ALLO_RULE = "frontier"
FRONTIER_WINDOW = 10

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.Session(profile_name="hm2p-agent").client("s3")
    return _S3


def _download_h5(key, retries=4):
    """Download an HDF5 from S3 into memory, retrying on transient errors.

    The container's S3 connection is intermittently flaky; without retries whole
    sessions silently drop. Returns None only after all attempts fail.
    """
    import time

    for attempt in range(retries):
        try:
            obj = _s3().get_object(Bucket=BUCKET, Key=key)
            return h5py.File(io.BytesIO(obj["Body"].read()), "r")
        except Exception as exc:  # noqa: BLE001
            log.debug("download failed %s (attempt %d): %s", key, attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    log.warning("download FAILED after %d attempts: %s", retries, key)
    return None


def load_metadata():
    base = Path(__file__).resolve().parent.parent / "metadata"
    animals = pd.read_csv(base / "animals.csv")
    animals["animal_id"] = animals["animal_id"].astype(str)
    exps = pd.read_csv(base / "experiments.csv")
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    return animals, exps


def sync_key(exp_id, animal_id):
    parts = exp_id.split("_")
    return f"sync/sub-{animal_id}/ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}/sync.h5"


def session_choice_stats(sync_f):
    need = ["x_maze", "y_maze", "light_on", "bad_behav", "speed_cm_s"]
    for r in need:
        if r not in sync_f:
            return None
    x = sync_f["x_maze"][:].astype(np.float64)
    y = sync_f["y_maze"][:].astype(np.float64)
    n = x.size
    light = sync_f["light_on"][:][:n].astype(bool)
    bad = sync_f["bad_behav"][:][:n].astype(bool)
    speed = sync_f["speed_cm_s"][:][:n].astype(np.float64)

    cell_idx = discretize_position_fast(x, y, MAZE).astype(np.int64)
    moving = ~bad & np.isfinite(speed) & (speed >= SPEED_THRESHOLD)
    cell_idx[~moving] = -1
    visit_cells, visit_frames = cell_sequence(cell_idx)
    if visit_cells.size < 3:
        return None

    events = extract_choice_events(
        visit_cells, visit_frames, MAZE, light,
        allo_rule=ALLO_RULE, frontier_window=FRONTIER_WINDOW,
    )
    if not events:
        return None

    rl, nl = conflict_follow_rate(events, "light")
    rd, nd = conflict_follow_rate(events, "dark")
    acc_l = rule_accuracies(events, "light")
    acc_d = rule_accuracies(events, "dark")
    return {
        "n_events": len(events),
        "conflict_allo_follow_light": rl,
        "conflict_allo_follow_dark": rd,
        "n_conflict_light": nl,
        "n_conflict_dark": nd,
        "ego_acc_light": acc_l["ego_acc"],
        "allo_acc_light": acc_l["allo_acc"],
        "ego_acc_dark": acc_d["ego_acc"],
        "allo_acc_dark": acc_d["allo_acc"],
    }


def _paired(light_vals, dark_vals, label):
    a = np.asarray(light_vals, float)
    b = np.asarray(dark_vals, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 2 or np.all(a - b == 0):
        return {"label": label, "n": n, "p": np.nan, "median_diff": np.nan,
                "rank_biserial": np.nan, "n_light_gt_dark": int(np.sum(a > b))}
    diff = a - b
    nz = diff[diff != 0]
    w, p = stats.wilcoxon(a, b, alternative="two-sided")
    ranks = stats.rankdata(np.abs(nz))
    rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {"label": label, "n": n, "p": float(p), "median_diff": float(np.median(diff)),
            "rank_biserial": rb, "med_light": float(np.median(a)), "med_dark": float(np.median(b)),
            "n_light_gt_dark": int(np.sum(a > b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-conflict", type=int, default=5,
                    help="min conflict trials per condition for a session to count")
    ap.add_argument("--allo-rule", choices=["myopic", "frontier"], default="frontier",
                    help="allocentric rule: myopic recency or distance-to-frontier")
    ap.add_argument("--frontier-window", type=int, default=10,
                    help="visits-since-last-visit beyond which a cell is frontier")
    ap.add_argument("--output", type=Path, default=Path("results/controller_switch"))
    args = ap.parse_args()
    global ALLO_RULE, FRONTIER_WINDOW
    ALLO_RULE = args.allo_rule
    FRONTIER_WINDOW = args.frontier_window

    animals, exps = load_metadata()
    valid = exps[exps["exclude"].astype(str).str.strip() != "1"]
    if args.limit:
        valid = valid.head(args.limit)

    rows = []
    for _, exp in valid.iterrows():
        exp_id, animal_id = exp["exp_id"], exp["animal_id"]
        f = _download_h5(sync_key(exp_id, animal_id))
        if f is None:
            continue
        try:
            st = session_choice_stats(f)
        finally:
            f.close()
        if st is None:
            continue
        ct = str(animals.loc[animals["animal_id"] == animal_id, "celltype"].iloc[0]) \
            if (animals["animal_id"] == animal_id).any() else ""
        rows.append({"exp_id": exp_id, "animal_id": animal_id, "celltype": ct, **st})
        log.info("  %s: events=%d conflict L/D=%d/%d allo-follow L/D=%.2f/%.2f",
                 exp_id, st["n_events"], st["n_conflict_light"], st["n_conflict_dark"],
                 st["conflict_allo_follow_light"], st["conflict_allo_follow_dark"])

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output / "controller_switch.csv", index=False)

    # Headline: allocentric-follow on conflict trials, light vs dark — only
    # sessions with enough conflict trials in BOTH conditions.
    ok = df[(df["n_conflict_light"] >= args.min_conflict)
            & (df["n_conflict_dark"] >= args.min_conflict)]
    head = _paired(ok["conflict_allo_follow_light"], ok["conflict_allo_follow_dark"],
                   "conflict allo-follow (light vs dark)")
    allo_acc = _paired(df["allo_acc_light"], df["allo_acc_dark"], "allocentric accuracy")
    ego_acc = _paired(df["ego_acc_light"], df["ego_acc_dark"], "egocentric accuracy")

    def fmt(t):
        if not np.isfinite(t.get("p", np.nan)):
            return f"- **{t['label']}**: N={t['n']} (insufficient)"
        return (f"- **{t['label']}**: N={t['n']}, median light={t.get('med_light', float('nan')):.3f} "
                f"dark={t.get('med_dark', float('nan')):.3f}, Wilcoxon p={t['p']:.4f}, "
                f"median(light-dark)={t['median_diff']:.3f}, rank-biserial={t['rank_biserial']:.3f}, "
                f"{t['n_light_gt_dark']}/{t['n']} light>dark")

    lines = [
        "# Controller switch — junction choice rules, light vs dark (behaviour only)",
        "",
        f"Sessions analysed: {len(df)} (>= {args.min_conflict} conflict trials/condition "
        f"for the headline: {len(ok)}).",
        "",
        "Conflict trial = junction where the egocentric (alternation/forward) and "
        "allocentric (least-recently-visited) rules predict different arms. The "
        "headline asks how often the allocentric arm is taken, light vs dark; a drop "
        "in dark = the world-based rule is vision-gated.",
        "",
        "## Headline",
        fmt(head),
        "",
        "## Standalone rule accuracy (predicting the actual choice)",
        fmt(allo_acc),
        fmt(ego_acc),
        "",
        "## Caveats",
        "- Per-session paired; conflict trials defined identically in both conditions, "
        "but junction-identity sampling is not yet stratified (planned). Treat as a "
        "first pass.",
        "- Recency is whole-session (memory does not reset across epochs).",
        "- Allocentric rule undecidable when several arms are equally unvisited; those "
        "events are excluded from conflict trials by construction.",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
