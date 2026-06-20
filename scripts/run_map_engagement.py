#!/usr/bin/env python3
"""Map-engagement: population-vector consistency across maze-cell revisits,
light vs dark (paired, within session).

Tests whether the RSP population re-instantiates the same state on repeat visits
to the same maze cell (the spatial map being "engaged"), and whether that
consistency weakens in darkness — the neural correlate of the behavioural
finding that exploration goes from directed (light) to random-walk-like (dark).
See docs/plan-map-engagement-neural.md.

Design
------
- Unit = session; light vs dark PAIRED within session; Wilcoxon signed-rank.
- A visit = a contiguous run of valid moving frames in one maze cell, summarised
  by its mean z-scored population vector (soma ROIs).
- Per condition: within-cell minus across-cell mean pairwise correlation
  (debiased for global drift / arousal).
- Sampling matched across conditions: same number of cells and visits/cell
  (cap = min eligible cells over the two conditions), so the dark coverage drop
  does not drive the estimate.

Usage
-----
    python scripts/run_map_engagement.py                 # full run
    python scripts/run_map_engagement.py --limit 3       # quick
    python scripts/run_map_engagement.py --k-visits 3 --n-boot 50
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

from hm2p.analysis.map_engagement import consistency_debiased, extract_visit_vectors  # noqa: E402
from hm2p.maze.discretize import discretize_position_fast  # noqa: E402
from hm2p.maze.topology import build_rose_maze  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("map_engagement")

BUCKET = "hm2p-derivatives"
SPEED_THRESHOLD = 2.5
MIN_SOMA = 5
MAZE = build_rose_maze()

_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3

        _S3 = boto3.Session(profile_name="hm2p-agent").client("s3")
    return _S3


def _download_h5(key: str) -> h5py.File | None:
    try:
        obj = _s3().get_object(Bucket=BUCKET, Key=key)
        return h5py.File(io.BytesIO(obj["Body"].read()), "r")
    except Exception as exc:  # noqa: BLE001
        log.debug("download failed %s: %s", key, exc)
        return None


def load_metadata():
    base = Path(__file__).resolve().parent.parent / "metadata"
    animals = pd.read_csv(base / "animals.csv")
    animals["animal_id"] = animals["animal_id"].astype(str)
    exps = pd.read_csv(base / "experiments.csv")
    exps["animal_id"] = exps["exp_id"].str.split("_").str[-1]
    return animals, exps


def sync_key(exp_id: str, animal_id: str) -> str:
    parts = exp_id.split("_")
    sub = f"sub-{animal_id}"
    ses = f"ses-{parts[0]}T{parts[1]}{parts[2]}{parts[3]}"
    return f"sync/{sub}/{ses}/sync.h5"


def session_consistency(sync_f, signal_name, k_visits, n_boot, rng):
    """Return (debiased_light, debiased_dark, info) or None."""
    need = ["roi_types", "x_maze", "y_maze", "light_on", "bad_behav", "speed_cm_s", signal_name]
    for r in need:
        if r not in sync_f:
            return None
    roi_types = sync_f["roi_types"][:]
    soma = np.where(roi_types == 0)[0]
    if soma.size < MIN_SOMA:
        return None
    sig = sync_f[signal_name][:][soma].astype(np.float64)  # (n_soma, n_frames)
    n_frames = sig.shape[1]
    # z-score each ROI over the session
    mu = sig.mean(axis=1, keepdims=True)
    sd = sig.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    sigz = (sig - mu) / sd

    x = sync_f["x_maze"][:][:n_frames].astype(np.float64)
    y = sync_f["y_maze"][:][:n_frames].astype(np.float64)
    light = sync_f["light_on"][:][:n_frames].astype(bool)
    bad = sync_f["bad_behav"][:][:n_frames].astype(bool)
    speed = sync_f["speed_cm_s"][:][:n_frames].astype(np.float64)

    cell_idx = discretize_position_fast(x, y, MAZE)
    moving = ~bad & np.isfinite(speed) & (speed >= SPEED_THRESHOLD)

    vc_l, vv_l = extract_visit_vectors(sigz, cell_idx, moving & light)
    vc_d, vv_d = extract_visit_vectors(sigz, cell_idx, moving & ~light)

    def eligible(vc):
        if vc.size == 0:
            return 0
        u, c = np.unique(vc, return_counts=True)
        return int((c >= k_visits).sum())

    cap = min(eligible(vc_l), eligible(vc_d))
    if cap < 2:
        return None

    res_l = consistency_debiased(vc_l, vv_l, k_visits=k_visits, n_cells_cap=cap, n_boot=n_boot, rng=rng)
    res_d = consistency_debiased(vc_d, vv_d, k_visits=k_visits, n_cells_cap=cap, n_boot=n_boot, rng=rng)
    info = {
        "n_soma": int(soma.size),
        "n_visits_light": int(vc_l.size),
        "n_visits_dark": int(vc_d.size),
        "n_cells_used": cap,
        "within_light": res_l["within"],
        "across_light": res_l["across"],
        "within_dark": res_d["within"],
        "across_dark": res_d["across"],
    }
    return res_l["debiased"], res_d["debiased"], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="dff", choices=["dff", "events"])
    ap.add_argument("--k-visits", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("results/map_engagement"))
    args = ap.parse_args()

    animals, exps = load_metadata()
    valid_exps = exps[exps["exclude"].astype(str).str.strip() != "1"]
    if args.limit:
        valid_exps = valid_exps.head(args.limit)

    rng = np.random.default_rng(args.seed)
    rows = []
    deb_l, deb_d = [], []
    for _, exp in valid_exps.iterrows():
        exp_id = exp["exp_id"]
        animal_id = exp["animal_id"]
        f = _download_h5(sync_key(exp_id, animal_id))
        if f is None:
            continue
        try:
            out = session_consistency(f, args.signal, args.k_visits, args.n_boot, rng)
        finally:
            f.close()
        if out is None:
            log.info("  skip %s (insufficient revisits/cells)", exp_id)
            continue
        dl, dd, info = out
        if not (np.isfinite(dl) and np.isfinite(dd)):
            continue
        celltype = str(animals.loc[animals["animal_id"] == animal_id, "celltype"].iloc[0]) \
            if (animals["animal_id"] == animal_id).any() else ""
        deb_l.append(dl)
        deb_d.append(dd)
        rows.append({"exp_id": exp_id, "animal_id": animal_id, "celltype": celltype,
                     "debiased_light": dl, "debiased_dark": dd, **info})
        log.info("  %s: light=%.3f dark=%.3f (cells=%d, soma=%d)",
                 exp_id, dl, dd, info["n_cells_used"], info["n_soma"])

    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output / "map_engagement.csv", index=False)

    a, b = np.array(deb_l), np.array(deb_d)
    n = len(a)
    if n >= 2:
        diff = a - b  # light - dark
        nz = diff[diff != 0]
        if nz.size >= 1:
            w, p = stats.wilcoxon(a, b, alternative="two-sided")
            ranks = stats.rankdata(np.abs(nz))
            rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        else:
            w, p, rb = np.nan, np.nan, np.nan
    else:
        w, p, rb = np.nan, np.nan, np.nan

    lines = [
        "# Map engagement — population-vector consistency across revisits",
        "",
        f"**Signal:** {args.signal} | **k visits:** {args.k_visits} | "
        f"**boot:** {args.n_boot} | **N sessions:** {n}",
        "",
        "within-cell minus across-cell mean pairwise correlation of per-visit "
        "population vectors; sampling matched (equal cells + visits/cell) across "
        "light/dark. Paired Wilcoxon, light vs dark. Positive (light-dark) => map "
        "more engaged in light.",
        "",
        f"- Light debiased consistency: median {np.median(a):.4f}" if n else "- no sessions",
        f"- Dark debiased consistency:  median {np.median(b):.4f}" if n else "",
        f"- Wilcoxon p={p:.4f}; median(light-dark)={np.median(a - b):.4f}; "
        f"rank-biserial={rb:.3f}; {int(np.sum(a > b))}/{n} sessions light>dark" if n else "",
    ]
    report = "\n".join([ln for ln in lines if ln != ""])
    (args.output / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
