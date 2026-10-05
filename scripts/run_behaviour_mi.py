#!/usr/bin/env python3
"""Single-cell mutual information with maze behaviours, all cells (``hm2p.analysis.behaviour_mi``).

Per session (from a local cache of pickled session arrays) and per cell, the
shift-debiased MI between binned activity and:

- ``inout``: frames inside duration/speed-matched runs into (1) vs out of (0)
  dead ends (the population decoder's runs), plain and conditional on head
  direction (8 sectors);
- ``position``: maze cell during running (speed >= 2.5 cm/s), plain and
  conditional on head direction;
- ``light``: lights on vs off (all valid frames);
- ``running``: running (>= 2.5 cm/s) vs still (< 1 cm/s), as a reference scale;
- ``hd``: head direction in 8 sectors during running, plain and conditional on
  maze cell (the mirror of position given head direction);
- speed controls: ``given_speed`` (running-speed tertile), ``given_hd_speed`` and
  ``given_position_speed`` (conjunctive gain beyond speed is given_hd_speed vs
  given_speed for position, and given_position_speed vs given_speed for head
  direction), because running speed itself varies with place and direction;
- ``position@light`` / ``position@dark`` and ``hd@light`` / ``hd@dark``: the same
  restricted to frames in one light condition.

Output: ``results/behaviour_mi/cells_<signal>.csv``.

Usage
-----
    python scripts/run_behaviour_mi.py --cache <dir>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hm2p.analysis.behaviour_mi import cell_label_mi, frames_from_windows, hd_sectors
from hm2p.analysis.population_maze import PopDecParams, build_samples, draw_shifts, session_cells
from hm2p.maze.topology import build_rose_maze

REPO = Path(__file__).resolve().parent.parent
RUN_CM_S, STILL_CM_S = 2.5, 1.0


def session_labels(d: dict[str, Any], n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-frame labels (-1 = unused) for every behaviour."""

    def g(k: str) -> np.ndarray:
        return np.asarray(d[k], dtype=float)[:n]

    valid = ~np.asarray(d["bad_behav"], dtype=bool)[:n]
    light = np.asarray(d["light_on"], dtype=bool)[:n]
    spd = g("speed_cm_s")
    maze = build_rose_maze()
    samples, _ = build_samples(
        g("x_maze"),
        g("y_maze"),
        spd,
        light,
        valid,
        float(d["fps"]),
        PopDecParams(),
        maze,
        np.random.default_rng(seed),
    )
    s = samples["inout"]
    inout = (
        np.full(n, -1, dtype=np.int64)
        if s.reason
        else frames_from_windows(n, s.starts, s.ends, s.labels)
    )
    cells = session_cells(g("x_maze"), g("y_maze"), valid, maze)
    position = np.where((spd >= RUN_CM_S) & (cells >= 0), cells, -1)
    running = np.full(n, -1, dtype=np.int64)
    running[valid & (spd >= RUN_CM_S)] = 1
    running[valid & (spd < STILL_CM_S)] = 0
    lightlab = np.where(valid, light.astype(np.int64), -1)
    hd = hd_sectors(g("hd_deg"))
    hd_run = np.where((spd >= RUN_CM_S) & valid & (hd >= 0), hd, -1)
    out = {
        "inout": inout,
        "position": position,
        "light": lightlab,
        "running": running,
        "hd": hd_run,
    }
    for cname, m in (("light", light), ("dark", ~light)):
        out[f"position@{cname}"] = np.where(m, position, -1)
        out[f"hd@{cname}"] = np.where(m, hd_run, -1)
    return out


def session_mi(
    d: dict[str, Any], n_shuffles: int = 200, signal: str = "spikes", seed: int = 0
) -> pd.DataFrame:
    """Per-cell MI rows for one session."""
    sig = np.nan_to_num(np.asarray(d[signal], dtype=float))
    n = sig.shape[1]
    labels = session_labels(d, n, seed)
    hd = hd_sectors(np.asarray(d["hd_deg"], dtype=float)[:n])
    pos = labels["position"]
    spd = np.asarray(d["speed_cm_s"], dtype=float)[:n]
    run = spd >= RUN_CM_S
    sbin = np.full(n, -1, dtype=np.int64)
    if run.any():  # speed tertiles among running frames
        edges = np.quantile(spd[run], [1 / 3, 2 / 3])
        sbin[run] = np.searchsorted(edges, spd[run], side="right")
    hd_speed = np.where((hd >= 0) & (sbin >= 0), hd * 3 + sbin, -1)
    pos_speed = np.where((pos >= 0) & (sbin >= 0), pos * 3 + sbin, -1)
    rng = np.random.default_rng(seed)
    shifts = draw_shifts(n, n_shuffles, int(round(30.0 * float(d["fps"]))), rng)[1:]
    out = []
    for name, lab in labels.items():
        if (lab >= 0).sum() < 200 or np.unique(lab[lab >= 0]).size < 2:
            continue
        base = name.split("@")[0]
        variants = (
            ("plain", None),
            ("given_hd", hd),
            ("given_position", pos),
            ("given_speed", sbin),
            ("given_hd_speed", hd_speed),
            ("given_position_speed", pos_speed),
        )
        for cond_name, cond in variants:
            if cond_name in ("given_hd", "given_hd_speed") and base not in ("inout", "position"):
                continue
            if cond_name in ("given_position", "given_position_speed") and base != "hd":
                continue
            if cond_name == "given_speed" and base not in ("position", "hd"):
                continue
            t = cell_label_mi(sig, lab, shifts, cond=cond)
            t["behaviour"] = name
            t["variant"] = cond_name
            t["n_frames"] = int((lab >= 0).sum())
            out.append(t)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    res["exp_id"] = d["meta"]["exp_id"]
    res["animal_id"] = str(d["meta"]["animal_id"])
    return res


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - file I/O entry point
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--signal", default="spikes", choices=["spikes", "dff"])
    ap.add_argument("--n-shuffles", type=int, default=200)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "behaviour_mi")
    args = ap.parse_args(argv)
    rows = []
    for f in sorted(args.cache.glob("*.pkl")):
        with open(f, "rb") as fh:
            t = session_mi(pickle.load(fh), args.n_shuffles, args.signal)
        rows.append(t)
        print("done", f.stem, len(t), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.out / f"cells_{args.signal}.csv", index=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
