#!/usr/bin/env python3
"""Spike-frequency adaptation of Penk+ vs Penk- RSP L2/3 cells from IV sweeps.

Reads each patched cell's current-step (IV) WaveSurfer file, computes
adaptation on the max-spike sweep and on the lowest-current sweep with at
least ``--target-spikes`` spikes (``hm2p.patching.adaptation``), and compares
the groups. Penk+ and Penk- cells were largely recorded in different mice
(one mouse contributed both), so the primary test permutes labels at the
animal level; cell-level Mann-Whitney is reported as descriptive only, and
the single mixed mouse is reported separately as the only within-animal
contrast.

Usage
-----
    python scripts/run_patching_adaptation.py --dry-run
    python scripts/run_patching_adaptation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hm2p.patching.adaptation import cell_adaptation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("patching_adaptation")

REPO = Path(__file__).resolve().parent.parent
EPHYS_ROOT = Path("/data/patching/ephys")
META = REPO / "metadata" / "patching"
OUT = REPO / "results" / "patching" / "celltype_programme"
METRICS = [
    "max_n_spikes",
    "max_adaptation_index",
    "max_sfa_ratio",
    "max_isi_cv",
    "max_first_isi_ms",
    "max_early_late_rate_ratio",
    "matched_adaptation_index",
    "matched_sfa_ratio",
    "matched_isi_cv",
    "matched_first_isi_ms",
    "matched_early_late_rate_ratio",
    "matched_current",
    "fi_slope_spikes_per_pa",
]


def cell_table(meta_dir: Path = META) -> pd.DataFrame:
    """Patched cells with ephys, joined to the slice date that names the ephys folder."""
    cells = pd.read_csv(meta_dir / "cells.csv", encoding="latin-1")
    animals = pd.read_csv(meta_dir / "animals.csv", dtype={"date_slice": str})
    cells = cells[cells["ephys_id"].astype(str).str.strip().str.startswith("SW")]
    out = cells.merge(animals[["animal_id", "date_slice"]], on="animal_id", how="left")
    return out[["cell_index", "animal_id", "cell_type", "ephys_id", "date_slice"]].reset_index(
        drop=True
    )


def find_iv_file(root: Path, date_slice: str, ephys_id: str) -> Path | None:
    """The IV file for a cell: top-level ``IV*.h5`` in its folder, else ``IV/IV*.h5``."""
    folder = root / str(date_slice) / str(ephys_id).strip()
    for pattern in ("IV*.h5", "IV/IV*.h5"):
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[-1]
    return None


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta P(a>b) - P(a<b); NaN if either group is empty."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / diff.size)


def compare(df: pd.DataFrame, n_perms: int, seed: int) -> pd.DataFrame:
    """Per metric: medians, Cliff's delta, descriptive cell MWU, animal-level permutation p."""
    from scipy import stats

    from hm2p.patching.statistics import cluster_permutation_comparison

    perm = cluster_permutation_comparison(
        df,
        METRICS,
        group_col="cell_type",
        animal_col="animal_id",
        n_perms=n_perms,
        rng=np.random.default_rng(seed),
        within_animal=False,
    ).set_index("metric")
    rows = []
    mixed = df.groupby("animal_id")["cell_type"].nunique()
    mixed_animals = mixed[mixed > 1].index.tolist()
    for m in METRICS:
        pos = df.loc[df["cell_type"] == "penkpos", m].to_numpy(dtype=float)
        neg = df.loc[df["cell_type"] == "penkneg", m].to_numpy(dtype=float)
        pf, nf = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
        p_cell = (
            float(stats.mannwhitneyu(pf, nf, alternative="two-sided").pvalue)
            if pf.size >= 3 and nf.size >= 3
            else np.nan
        )
        row: dict[str, Any] = {
            "metric": m,
            "n_penkpos": int(pf.size),
            "n_penkneg": int(nf.size),
            "median_penkpos": float(np.median(pf)) if pf.size else np.nan,
            "median_penkneg": float(np.median(nf)) if nf.size else np.nan,
            "cliffs_delta": cliffs_delta(pos, neg),
            "cell_mwu_p_descriptive": p_cell,
            "animal_perm_p": float(perm.loc[m, "p_perm"]) if m in perm.index else np.nan,
        }
        for a in mixed_animals:
            sub = df[df["animal_id"] == a]
            row[f"within_{a}_delta"] = cliffs_delta(
                sub.loc[sub["cell_type"] == "penkpos", m].to_numpy(dtype=float),
                sub.loc[sub["cell_type"] == "penkneg", m].to_numpy(dtype=float),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ephys-root", type=Path, default=EPHYS_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--target-spikes", type=int, default=6)
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - reads raw data
    args = build_parser().parse_args(argv)
    from hm2p.patching.io import load_wavesurfer
    from hm2p.patching.protocols import extract_iv

    cells = cell_table()
    cells["iv_file"] = [
        find_iv_file(args.ephys_root, r.date_slice, r.ephys_id) for r in cells.itertuples()
    ]
    log.info("%d cells with ephys; %d IV files found", len(cells), cells["iv_file"].notna().sum())
    if args.dry_run:
        print(cells.groupby("cell_type")["iv_file"].apply(lambda s: s.notna().sum()))
        return 0
    rows = []
    for r in cells.itertuples():
        if r.iv_file is None:
            continue
        try:
            ws = load_wavesurfer(r.iv_file)
            iv = extract_iv(ws)
            sr = float(ws["header"]["StimulationSampleRate"])
            res = cell_adaptation(iv.traces, iv.stim_vec, sr, target_spikes=args.target_spikes)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            log.warning("cell %s failed: %s", r.cell_index, exc)
            continue
        rows.append(
            {"cell_index": r.cell_index, "animal_id": r.animal_id, "cell_type": r.cell_type, **res}
        )
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "adaptation_cells.csv", index=False)
    rep = compare(df, args.n_perms, args.seed)
    rep.to_csv(args.out / "adaptation_comparison.csv", index=False)
    (args.out / "adaptation_meta.json").write_text(
        json.dumps({"n_cells": len(df), "target_spikes": args.target_spikes}, indent=2)
    )
    print(rep.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
